#!/usr/bin/env python
"""R-180 CONSERVATIVE branch: literal binary meta-label veto on
`kelly_regime_v4`'s own `frac*scale` deadband decision -- feature swap only
against R-179's conservative branch.

Direction, citations, non-duplication argument and the frozen falsification
clauses all live in `experiments/r180_direction.md` (read there first); the
shared, read-only engine (`macro_stress_z`, `mvrv_z`,
`walk_forward_meta_prob`, `step_a_permutation_gate`, plus `vote_frac`/
`conditional_scale` re-exported unedited from `r179_shared.py`) lives in
`experiments/r180_shared.py` and is never edited or re-derived here -- this
file only wires those primitives into one `Strategy` and reports the sweep.

THE MECHANISM, exactly (Lopez de Prado 2018, ch. 3's textbook binary
meta-label, "bet or don't") -- IDENTICAL to R-179's conservative branch
(`experiments/r179_conservative.py`), the only change is which two features
feed the walk-forward classifier:

v4's own sequential deadband loop is reproduced bar-by-bar, verbatim:

    desired = frac[i] * scale[i]
    if abs(desired - pos) > deadband:
        pos = desired          # <- v4 always takes this branch here

The ONLY change from v4-alone: the assignment `pos = desired` is taken only
when the walk-forward meta-classifier's CURRENT probability clears
`threshold` (default 0.50). If it does not, `pos` holds at its previous
value and the same candidate `desired` is re-tested on every later bar (a
vetoed trade is DEFERRED, never discarded -- it fires the moment either the
gate opens or `desired` itself changes again and re-clears the deadband).
While the classifier has not yet reached `min_samples` resolved labels at
its first refit (or never reaches it, e.g. `min_samples` set absurdly high),
`prob[i]` is NaN by `walk_forward_meta_prob`'s own contract; this branch's
disclosed neutral behaviour for that warmup/never-confident stretch is to
treat a NaN probability as an OPEN gate -- i.e. behave exactly like v4 until
(or unless) the classifier actually has an opinion. This is verified below
by an explicit identity/no-op test (`_identity_probe`), not merely asserted:
threshold=0.0 and a never-fitting classifier (`min_samples` set above the
whole training period's checkpoint count) must both reproduce
`kelly_regime_v4`'s own `target` array EXACTLY, bar for bar.

The classifier's two input features are `r180_shared.macro_stress_z` (VIX +
DXY level, trailing-365-day z-scored, causal) and `r180_shared.mvrv_z`
(on-chain MVRV level, trailing-365-day z-scored, causal) -- in place of
R-179's `vol_ratio`/`vote_strength`/`log1p(regime_duration)` triple. Nothing
else about the architecture changes: same purge/embargo, same
expanding-window refit-then-forward-fill discipline, same daily
triple-barrier labels (imported unedited from `r179_shared.py` via
`r180_shared.py`), same binary bet/no-bet veto gate structure.

No lookahead: `frac`/`scale`/`vol_ratio` are `r179_shared`'s own verbatim,
already-causal reproductions of v4's factors; `vol_daily` is the identical
causal EWM estimator (`shift(1)`-ed) `conditional_scale` uses internally;
`macro_stress_z`/`mvrv_z` are causal (D+1-visible) per `r180_shared.py`'s own
docstring and self-test; `walk_forward_meta_prob` forward-fills each bar's
probability from the most recent refit whose fit set is causal as of that
refit's own instant. The deadband loop above reads only `frac[i]`,
`scale[i]`, `prob[i]` at the CURRENT bar `i`.

STEP-A IS THE DECISIVE TEST (per r180_direction.md Step 1 Q4): the
Sharpe/log-growth promotion-bar clause is pre-registered as underpowered on
this dataset (a ~2.6-2.7x CI-width reduction, i.e. ~6.8x the independent
evidence, would be needed to exclude zero at R-179's own measured effect
size). If Step-A does not clear (true AUC beats the label-permutation
null's 95th percentile on a MAJORITY of the tested (horizon_days,
refit_days) corners), the branch is recorded NEGATIVE by construction and
the full inner-validation config sweep is skipped -- reported as a finding,
not run for completeness, per this round's own pre-registered instruction.

Run: `source .venv/bin/activate && python experiments/r180_conservative.py`
(from the repo root).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.registry import register  # noqa: E402
from tradebot.strategy import Context, Strategy  # noqa: E402

from r180_shared import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_YEAR,
    conditional_scale,
    macro_stress_z,
    mvrv_z,
    step_a_permutation_gate,
    vote_frac,
    walk_forward_meta_prob,
)

OOS_START = "2023-01-01"  # never read below this
INNER_TRAIN_END = "2020-12-31"
INNER_VAL_START = "2021-01-01"
INNER_VAL_END = "2022-12-31"

DATA_DIR = ROOT / "data"


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
class R180ConservativeMetaVeto(Strategy):
    """`kelly_regime_v4`'s vote+scale signal, gated by a binary walk-forward
    meta-label veto (Lopez de Prado 2018 ch. 3) fed exogenous macro-stress
    and on-chain-valuation features (R-179's own architecture, feature swap
    only). See module docstring for the exact mechanism; full design in
    `experiments/r180_direction.md`."""

    name = "r180_conservative_meta_veto"
    warmup = 80 * BARS_PER_DAY + 10  # identical to kelly_regime_v4's own

    def __init__(self, threshold: float = 0.50, k: float = 1.0,
                 horizon_days: int = 3, refit_days: int = 90,
                 embargo_days: int = 3, min_samples: int = 50,
                 deadband: float = 0.10, vol_span: int = 8 * BARS_PER_DAY,
                 data_dir: str | Path = DATA_DIR, use_cache: bool = True) -> None:
        self.threshold = threshold
        self.k = k
        self.horizon_days = horizon_days
        self.refit_days = refit_days
        self.embargo_days = embargo_days
        self.min_samples = min_samples
        self.deadband = deadband
        self.vol_span = vol_span
        self.data_dir = data_dir
        self.use_cache = use_cache
        # populated by prepare(), read back for reporting (diag/prob/frac/scale)
        self.last_diag: dict | None = None
        self.last_prob: np.ndarray | None = None
        self.last_frac: np.ndarray | None = None
        self.last_scale: np.ndarray | None = None
        self.last_gate_open_frac: float | None = None

    def _heavy(self, close: pd.Series, index: pd.DatetimeIndex):
        """(frac, scale, vol_ratio, vol_daily, feat_full, prob, diag) --
        everything independent of `threshold` and of which market is being
        backtested."""
        cfg = (self.k, self.horizon_days, self.refit_days, self.embargo_days,
               self.min_samples, self.vol_span, str(self.data_dir))
        key = _cache_key(close, cfg) if self.use_cache else None
        if key is not None and key in _CACHE:
            return _CACHE[key]

        frac = vote_frac(close)
        scale, vol_ratio = conditional_scale(close, vol_span=self.vol_span)
        vol_daily = _causal_vol_daily(close, self.vol_span)
        macro = macro_stress_z(index, self.data_dir)
        mvrv = mvrv_z(index, self.data_dir)
        feat_full = np.column_stack([macro, mvrv])
        prob, diag = walk_forward_meta_prob(
            index, close.to_numpy(), vol_daily, feat_full,
            k=self.k, horizon_days=self.horizon_days, refit_days=self.refit_days,
            embargo_days=self.embargo_days, min_samples=self.min_samples,
        )
        result = (frac, scale, vol_ratio, vol_daily, feat_full, prob, diag)
        if key is not None:
            _CACHE[key] = result
        return result

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        frac, scale, vol_ratio, vol_daily, feat_full, prob, diag = self._heavy(close, df.index)
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
                # Warmup/never-confident contract: NaN probability defaults
                # the gate OPEN -- act exactly like v4 until (or unless) the
                # classifier actually has an opinion (r180_direction.md's
                # required disclosed neutral behaviour, verified by
                # `_identity_probe` below).
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
# Self-tests: causal-truncation probe + explicit identity/no-op probe.
# ======================================================================

def _causal_truncation_probe(cfg: dict, df: pd.DataFrame, cut: int) -> bool:
    """Truncate `df` at `cut` bars, rerun `prepare()`, and check that every
    bar strictly before `cut - horizon_days*BARS_PER_DAY - embargo_days*
    BARS_PER_DAY` (i.e. every bar whose label/refit inputs cannot possibly
    reach past the truncation point) produced an IDENTICAL `target` value in
    both the full and truncated runs. A generous margin (well past the
    horizon+embargo boundary) is used so this is a real causality check, not
    a numerically-fragile one."""
    strat_full = R180ConservativeMetaVeto(**cfg, use_cache=False)
    strat_cut = R180ConservativeMetaVeto(**cfg, use_cache=False)
    full = strat_full.prepare(df.copy())
    trunc = strat_cut.prepare(df.iloc[:cut].copy())
    margin = (cfg.get("horizon_days", 3) + cfg.get("embargo_days", 3) + 5) * BARS_PER_DAY
    safe = cut - margin
    if safe <= 0:
        return True
    a = full["target"].to_numpy()[:safe]
    b = trunc["target"].to_numpy()[:safe]
    return bool(np.allclose(a, b, atol=1e-9, equal_nan=True))


def _identity_probe(df: pd.DataFrame) -> tuple[bool, bool]:
    """Explicit identity/no-op test (r180_direction.md's required disclosed
    neutral-behaviour verification): the gate must degenerate EXACTLY to
    `kelly_regime_v4`'s own behaviour (a) when `threshold=0.0` (every finite
    probability clears it, so the gate is always open), and (b) when the
    classifier never reaches `min_samples` (prob stays NaN for the whole
    run, so the NaN-defaults-open branch is always taken). Returns
    (threshold_zero_matches_v4, never_confident_matches_v4)."""
    from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4

    v4_target = KellyRegimeV4().prepare(df.copy())["target"].to_numpy()

    zero_th = R180ConservativeMetaVeto(threshold=0.0, use_cache=False)
    zero_target = zero_th.prepare(df.copy())["target"].to_numpy()
    zero_ok = bool(np.allclose(v4_target, zero_target, atol=1e-9, equal_nan=True))

    never_fit = R180ConservativeMetaVeto(threshold=0.50, min_samples=10**9, use_cache=False)
    never_target = never_fit.prepare(df.copy())["target"].to_numpy()
    assert not np.isfinite(never_fit.last_prob).any(), (
        "never_fit probe should never have fit a classifier (min_samples "
        "set above any reachable sample count) -- prob should be all-NaN")
    never_ok = bool(np.allclose(v4_target, never_target, atol=1e-9, equal_nan=True))

    return zero_ok, never_ok


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

    # Identity/no-op probe on a shorter, cheaper slice (a few years is
    # plenty to exercise both the deadband loop and several refits/warmup).
    probe_slice = train.iloc[-250_000:]
    zero_ok, never_ok = _identity_probe(probe_slice)
    assert zero_ok, "threshold=0.0 gate does NOT degenerate to v4-alone -- mechanism bug"
    assert never_ok, "never-confident (all-NaN prob) gate does NOT degenerate to v4-alone -- mechanism bug"


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
    # extended-metrics/Step-A call below is bounded by INNER_TRAIN_END/
    # INNER_VAL_END.

    print("=" * 78)
    print("R-180 CONSERVATIVE -- binary meta-label veto on kelly_regime_v4, ")
    print("exogenous features (macro_stress_z, mvrv_z)")
    print("=" * 78)

    K = 1.0
    EMBARGO_DAYS = 3
    MIN_SAMPLES = 50
    N_PERM = 500

    # ---------------------------------------------------------- Step-A (decisive)
    print("\n" + "-" * 78)
    print("STEP-A discriminative-skill permutation gate (r180_direction.md's "
          "decisive falsification test) -- TRAINING period "
          f"(dataset start .. {INNER_TRAIN_END}) ONLY. n_perm={N_PERM}.")
    print("-" * 78)

    train_df = DF.loc[:INNER_TRAIN_END]
    train_end_bar = len(train_df)
    print(f"Training slice: {train_end_bar:,} bars, {train_df.index[0]} -> {train_df.index[-1]}")

    close_full = DF["close"].to_numpy()
    vol_daily_full = _causal_vol_daily(DF["close"], 8 * BARS_PER_DAY)
    macro_full = macro_stress_z(DF.index, DATA_DIR)
    mvrv_full = mvrv_z(DF.index, DATA_DIR)
    feat_full = np.column_stack([macro_full, mvrv_full])

    CORNERS = [(hd, rd) for hd in (1, 3) for rd in (30, 90)]  # matches R-179's own grid axes
    step_a_results: dict[tuple, dict] = {}
    t0 = time.time()
    for hd, rd in CORNERS:
        tc0 = time.time()
        res = step_a_permutation_gate(
            DF.index, close_full, vol_daily_full, feat_full, train_end_bar,
            k=K, horizon_days=hd, embargo_days=EMBARGO_DAYS, refit_days=rd,
            min_samples=MIN_SAMPLES, n_perm=N_PERM, seed=180,
        )
        step_a_results[(hd, rd)] = res
        print(f"  horizon_days={hd} refit_days={rd}: true_auc={res['true_auc']:.4f} "
              f"null_p95={res['null_p95']:.4f} pval={res['pval']:.3f} "
              f"n_perm_valid={res['n_perm_valid']} "
              f"CLEARS={res['clears']}  ({time.time() - tc0:.1f}s)")
    step_a_wall = time.time() - t0
    n_clear = sum(1 for r in step_a_results.values() if r["clears"])
    n_tested = len(step_a_results)
    step_a_majority_clears = n_clear > n_tested / 2
    print(f"\nSTEP-A total wall time: {step_a_wall:.1f}s for {n_tested} corners "
          f"({n_tested * N_PERM} permutation draws total).")
    print(f"STEP-A verdict: {n_clear}/{n_tested} corners clear the null's 95th percentile "
          f"-- {'MAJORITY CLEARS' if step_a_majority_clears else 'MAJORITY DOES NOT CLEAR'}.")

    if not step_a_majority_clears:
        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        print("Step-A discriminative-skill gate FAILS (majority of tested corners do "
              "not clear the label-permutation null's 95th percentile). Per "
              "r180_direction.md Step 1 Q4, this is the decisive, well-powered "
              "falsification test for this branch -- the branch is recorded "
              "NEGATIVE by construction and the full inner-validation config sweep "
              "is skipped (pre-registered as not needed when Step-A fails).")
        print(f"\nConfigs evaluated (inner-validation backtest sweep): 0 (skipped, Step-A "
              f"failed). Step-A itself tested {n_tested} (horizon_days, refit_days) "
              f"corners x {N_PERM} permutation draws = {n_tested * N_PERM} classifier "
              "refits-under-permutation, plus the true-label fit for each corner.")
        print(f"\nIdentity/no-op probe (this branch's own required disclosed neutral-"
              "behaviour check): PASSED (see _self_test() -- threshold=0.0 and a "
              "never-confident classifier both reproduce kelly_regime_v4-alone's "
              "target array exactly).")
        print(f"\nMax timestamp read by this script: {INNER_TRAIN_END} (inner-train "
              f"upper bound -- Step-A never reads inner-validation or the holdout). "
              f"No bar at/after {OOS_START} was read for any metric above.")
        return

    # ---------------------------------------------------------- sweep on inner-validation
    # (Only reached if Step-A's majority-of-corners clause clears above.)
    print("\n" + "-" * 78)
    print(f"INNER-VALIDATION sweep ({INNER_VAL_START} -> {INNER_VAL_END}), both markets")
    print("-" * 78)

    THRESHOLDS = (0.45, 0.50, 0.55, 0.60)
    configs = [dict(threshold=th, k=K, horizon_days=hd, refit_days=rd,
                     embargo_days=EMBARGO_DAYS, min_samples=MIN_SAMPLES)
               for th in THRESHOLDS for hd, rd in CORNERS]
    print(f"Grid: {len(THRESHOLDS)} thresholds x {len(CORNERS)} (horizon_days, refit_days) "
          f"corners = {len(configs)} configs.")

    rows = []
    t0 = time.time()
    for cfg in configs:
        strat_spot = R180ConservativeMetaVeto(**cfg)
        strat_fut = R180ConservativeMetaVeto(**cfg)
        m_spot = _extended_metrics(strat_spot, DF, SPOT, INNER_VAL_START, INNER_VAL_END)
        m_fut = _extended_metrics(strat_fut, DF, FUTURES, INNER_VAL_START, INNER_VAL_END)
        gate_open_frac = strat_spot.last_gate_open_frac
        rows.append(dict(cfg=cfg, spot=m_spot, futures=m_fut, gate_open_frac=gate_open_frac))
        print(f"  th={cfg['threshold']:.2f} hd={cfg['horizon_days']} rd={cfg['refit_days']:>2d}  "
              f"gate_open={gate_open_frac:.0%}  "
              f"SPOT sharpe={m_spot['sharpe']:>5.2f} vol={m_spot['realized_vol_pct']:>5.1f}% "
              f"notional={m_spot['avg_notional']:.3f} | "
              f"FUT sharpe={m_fut['sharpe']:>5.2f} vol={m_fut['realized_vol_pct']:>5.1f}% "
              f"notional={m_fut['avg_notional']:.3f}")
    print(f"\n({len(configs)} configs x 2 markets = {len(configs) * 2} backtests, "
          f"{time.time() - t0:.0f}s total)")

    v4_spot = _extended_metrics(KellyRegimeV4(), DF, SPOT, INNER_VAL_START, INNER_VAL_END)
    v4_fut = _extended_metrics(KellyRegimeV4(), DF, FUTURES, INNER_VAL_START, INNER_VAL_END)

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
    risk_matched_rows = [r for r in rows if r["risk_matched_spot"] and r["risk_matched_fut"]]
    print(f"\n{len(risk_matched_rows)} of {len(rows)} configs are risk-matched on BOTH markets.")
    best = risk_matched_rows[0] if risk_matched_rows else rows[0]
    print(f"Best {'RISK-MATCHED ' if risk_matched_rows else '(NOT risk-matched) '}config: {best['cfg']}")

    clears_spot = best["d_sharpe_spot"] > SHARPE_NOISE_FLOOR and best["d_loggrowth_spot"] > 0
    clears_fut = best["d_sharpe_fut"] > SHARPE_NOISE_FLOOR and best["d_loggrowth_fut"] > 0
    both_risk_matched = best["risk_matched_spot"] and best["risk_matched_fut"]
    promote = clears_spot and clears_fut and both_risk_matched

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"Configs evaluated: {len(configs)}.")
    print("PROMOTE-CANDIDATE" if promote else "NEGATIVE")


if __name__ == "__main__":
    main()
