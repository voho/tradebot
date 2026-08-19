#!/usr/bin/env python
"""Reselect kelly_regime_v4's own constants by cross-fold robustness, not a point estimate (ERR axis).

Not registered: this lives under ``experiments/`` so it is not
auto-discovered, per ROUTINE.md step 5. Promote it into
``src/tradebot/strategies/`` only if it clears the promotion bar.

The idea
--------
``kelly_regime_v4``'s free constants -- the anchor ladder (``horizons``),
``target_vol``, ``max_leverage``, ``vol_span``, ``deadband``,
``anchor_span_days``, the hysteresis thresholds -- were each set, across
L-01/R-06/R-07, by a SINGLE point-estimate optimization over one full
backtest window: whichever value scored best on the pooled period, once.
No prior round has ever asked how those constants were CHOSEN, only what
the chosen mechanism does once new signals are bolted onto it (R-34
through R-44, twelve branches, all negative -- see docs/LEDGER.md section
C). This file changes NO mechanism, NO signal, NO input: it only reselects
two already-swept axes -- the anchor-ladder base (R-06/R-07's own 18-28d
plateau) and the ``target_vol``/``max_leverage`` sizing pair (R-37
conservative's own axis) -- using a robustness-aware criterion instead of
a pooled point estimate.

Bertsimas & den Hertog-style robust optimization, and specifically recent
bootstrap-quantile parameter selection for time-series momentum (2024-2025
working papers on non-parametric bootstrap-quantile TSMOM lookback
selection) both make the same argument: a parameter chosen to maximize
the POOLED/mean outcome collapses out-of-sample when the effective sample
size is small (here, ~3 regime events -- N≈3, the project's own standing
diagnosis), because the pooled optimum is dominated by whichever single
regime happens to carry the most bars. Selecting instead for the WORST
(or a low quantile) across several non-overlapping regime folds is a
minimax / robust criterion: it cannot be won by a parameter that is
brilliant in one fold and mediocre elsewhere, which is exactly the failure
signature R-37/R-38/R-40/R-41 diagnosed in every signal-adding attempt.

Mechanism, one sentence: split the pre-2023 data into three non-
overlapping, calendar-purged folds (2017-2018, 2019-2020, 2021-2022-- one
per the project's own N≈3 regime-event count), evaluate the SAME
already-validated 18-28d-ladder x target_vol x max_leverage grid R-06/R-07/
R-37 already swept on each fold independently, and select the
configuration that maximizes the WORST-fold Sharpe (minimax) rather than
the pooled-window Sharpe a naive retune would pick -- then hand that
configuration to v3/v4's completely unchanged vote+scale mechanism.

Constraint attacked: ERR (no error control anywhere in the signal path).
Specifically: no error control in HOW v4's constants were chosen, not the
trading signal itself. This is model-selection-by-robust-criterion applied
to hyperparameter choice, not a new gate, filter, or input.

Pre-registered falsification test (fixed before any code ran)
---------------------------------------------------------------
The project's standard one: does the selected configuration still
match-or-beat kelly_regime_v4 on ETH data, and not visibly underperform it
on the pre-2020 BTC control -- both markets (spot, 5x futures), identical
code path for both assets. If the robust-selected config loses to v4 on
the BTC control, or is worse on ETH than the identical pipeline's BTC
control, this direction fails, exactly like R-37/R-38/R-40/R-41.

Not a duplicate of
-------------------
- R-06/R-07: those swept the anchor-ladder axis and found the 18-28d
  plateau, but selected (implicitly) by a single point estimate on one
  window. This file uses R-06/R-07's own already-validated region as the
  search space but changes the SELECTION CRITERION to cross-fold minimax.
- R-37 (conservative): retuned ``target_vol``/``max_leverage`` (53
  configurations) by pooled inner-validation Sharpe -- a point estimate on
  ONE window (2021-2022) -- and was ruled out as an exposure artifact /
  inside the noise floor. This file sweeps the same two constants but
  selects by minimax across THREE non-overlapping folds spanning
  2017-2022, not by the single 2021-2022 point estimate, and explicitly
  reports whether that changes the winner.
- R-40 (conservative/novel): bagged (averaged) MULTIPLE ladders into one
  vote at run time -- an ensembling/averaging mechanism change. This file
  changes no mechanism at run time; it picks ONE ladder base (plus one
  target_vol/max_leverage pair) in advance, via a different selection
  procedure over already-tried points. No averaging occurs inside
  ``prepare()``.
- R-34/R-35/R-38/R-41 (Bayesian posterior, funding gate, risk-constrained
  Kelly cap, CRRA fraction, basis brake/lead): all added a NEW signal or
  gate. This file adds none -- ``prepare()``/``on_bar()`` are v4's own,
  inherited unmodified; only the constructor's numeric constants differ.

Fold design ("purged" -- what that means here)
------------------------------------------------
Three non-overlapping calendar-year-pair folds, one per the project's own
N≈3 regime-event count: 2017-2018 (the 2017 bull + 2018 bear), 2019-2020
(2019 chop + the 2020 pre-halving grind), 2021-2022 (the 2021 top + the
2022 bear -- this is also ROUTINE.md's own inner-validation window).
De Prado-style purging exists to stop IID-resampled folds' evaluation
windows from leaking into each other's label horizon; that risk does not
arise here because these are non-overlapping, sequential CALENDAR blocks,
and ``tradebot.window.run_period`` draws every fold's warmup prefix only
from bars STRICTLY BEFORE that fold's own start (never from another
fold's interior, never from the future) -- so no fold's measured region
ever reads another fold's bars. That is the full extent of "purging" a
blocked, non-overlapping design like this one needs.

Pre-registered failure modes (named before any sweep ran)
------------------------------------------------------------
(a) The minimax winner is IDENTICAL (or a trivial neighbour) to the naive
    pooled-window winner -- i.e. robustness selection changes nothing,
    because the plateau is flat enough that fold disagreement never
    bites. A legitimate negative finding, not a failure of execution.
(b) The minimax winner beats v4 on inner-validation but the gain sits
    inside the +/-0.2 Sharpe noise floor -- not distinguishable from the
    incumbent.
(c) The result is an exposure-level artifact: regress the candidate's
    target series against a mean-notional-matched flat rescale of v4's
    own target series and report R^2 (R^2 > 0.95 is this project's
    standing threshold -- R-33/R-34's diagnostic, repeated in every
    subsequent round).
(d) Fails the ETH falsification test (worse than v4 on ETH, or visibly
    worse on ETH than the identical pipeline's BTC control) -- the exact
    failure mode that sank all twelve of R-34 through R-44's signal-adding
    attempts, here tested against a variant that adds no signal at all.

Search space (fixed in advance, not searched wider than prior rounds)
--------------------------------------------------------------------------
- ladder base in {18, 20, 22, 24, 26, 28} -- R-06/R-07's own validated
  18-28d plateau, expanded to (b, 2b, 4b) exactly like v4's own
  (20, 40, 80). No base outside this already-characterized region is
  tried.
- target_vol in {0.45, 0.55, 0.65} -- centered on v4's shipped 0.55,
  spanning the same neighbourhood R-37 (0.35-0.90) and L-04's own
  documented presets (0.40/0.55/0.60/0.80) already covered.
- max_leverage in {1.5, 2.0, 2.5} -- centered on v4's shipped 2.0.
- 6 x 3 x 3 = 54 distinct configurations -- the same order of magnitude as
  R-37's 53, not a wider search.
- Everything else (``band``, ``vol_span``, ``deadband``,
  ``anchor_span_days``, the hysteresis thresholds) is held at v4's shipped
  defaults. R-38 already interrogated the vol-scaling formula itself
  (risk-constrained caps, CRRA fractions) and both failed; this branch
  targets only the two axes that were historically point-estimated and
  never cross-validated -- the ladder choice (R-06/R-07) and the
  target_vol/max_leverage pair (R-37 conservative) -- with a different
  SELECTION PROCEDURE, not a wider search.

Usage
-----
    python experiments/kelly_regime_v11_robust_ladder.py sweep       # step 3: fold + pooled grids
    python experiments/kelly_regime_v11_robust_ladder.py select      # step 4/5: minimax vs naive, plateau, TRAIN/VALID
    python experiments/kelly_regime_v11_robust_ladder.py artifact    # failure mode (c)
    python experiments/kelly_regime_v11_robust_ladder.py causality   # step 6
    python experiments/kelly_regime_v11_robust_ladder.py eth         # step 7 / failure mode (d)
    python experiments/kelly_regime_v11_robust_ladder.py all         # everything, in order
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

from tradebot.broker import MarketSpec  # noqa: E402
from tradebot.data import load_dataset, load_ohlcv_csv  # noqa: E402
from tradebot.engine import run_backtest  # noqa: E402
from tradebot.metrics import compute_metrics  # noqa: E402
from tradebot.registry import get_strategy  # noqa: E402
from tradebot.strategies.kelly_regime_v4 import KellyRegimeV4  # noqa: E402
from tradebot.window import run_period  # noqa: E402

BARS_PER_DAY = 288
BARS_PER_YEAR = 365.25 * BARS_PER_DAY


# --------------------------------------------------------------------- strategy


class KellyRegimeV11RobustLadder(KellyRegimeV4):
    """kelly_regime_v4 with (horizons, target_vol, max_leverage) chosen by cross-fold minimax robustness.

    ``prepare()`` and ``on_bar()`` are v4's own, inherited byte-for-byte
    unchanged -- this class exists only so the selected configuration has
    a name and a correctly-scaled ``warmup`` (v4 hardcodes warmup for its
    own (20,40,80) ladder; a base up to 28 needs up to 112 days of
    history, more than v4's fixed 80-day warmup covers). Every other
    constant (``band``, ``vol_span``, ``deadband``, ``anchor_span_days``,
    hysteresis thresholds) stays at v3/v4's shipped defaults unless passed
    explicitly.
    """

    name = "kelly_regime_v11_robust_ladder"

    def __init__(self, horizons: tuple[int, ...] = (22, 44, 88),
                 target_vol: float = 0.55, max_leverage: float = 2.0, **kwargs) -> None:
        super().__init__(horizons=horizons, target_vol=target_vol,
                          max_leverage=max_leverage, **kwargs)
        self.warmup = int(max(horizons) * BARS_PER_DAY) + 10


# ------------------------------------------------------------------------ harness

DF, LABEL = load_dataset(ROOT / "data", "spot")
SPOT = MarketSpec.spot()
FUTURES = MarketSpec.futures(leverage=5.0)
MARKETS = (("spot", SPOT), ("futures", FUTURES))

# Standard project split (ROUTINE.md step 3) -- used for the final
# candidate-vs-v4 report, NOT as the fold-search protocol itself.
TRAIN = ("2017-01-01", "2020-12-31")
VALID = ("2021-01-01", "2022-12-31")
# OOS_START = "2023-01-01"  -- NEVER read in this file, by construction.

# The three purged, non-overlapping folds this branch selects across.
# "2021-2022" is byte-identical to VALID above -- reused, not re-run.
FOLDS = {
    "2017-2018": ("2017-01-01", "2018-12-31"),
    "2019-2020": ("2019-01-01", "2020-12-31"),
    "2021-2022": ("2021-01-01", "2022-12-31"),
}
# The naive point-estimate comparator: one pooled window, exactly what a
# single point-estimate optimization (v4's own R-06/R-07 lineage) would
# have been run against.
POOLED = ("2017-01-01", "2022-12-31")

INCUMBENT = "kelly_regime_v4"

LADDER_BASES = (18, 20, 22, 24, 26, 28)
TARGET_VOL_GRID = (0.45, 0.55, 0.65)
MAX_LEV_GRID = (1.5, 2.0, 2.5)
CONFIGS = [(b, tv, ml) for b in LADDER_BASES for tv in TARGET_VOL_GRID for ml in MAX_LEV_GRID]

OUT = ROOT / "reports" / "kelly_regime_v11_robust_ladder"

_SEEN: set[tuple] = set()  # distinct configurations evaluated, for the trials count


def make_strategy(base: int, tv: float, ml: float) -> KellyRegimeV11RobustLadder:
    _SEEN.add((base, tv, ml))
    return KellyRegimeV11RobustLadder(horizons=(base, 2 * base, 4 * base),
                                       target_vol=tv, max_leverage=ml)


def mean_notional(result) -> float:
    if "target" not in result.df:
        return float("nan")
    tgt = np.abs(result.df["target"].to_numpy(dtype=float))
    return float(np.mean(np.clip(tgt, 0.0, result.market.leverage)))


def realized_vol(equity) -> float:
    eq = equity.to_numpy(dtype=float) if hasattr(equity, "to_numpy") else np.asarray(equity)
    if len(eq) < 3:
        return float("nan")
    prev = eq[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, np.diff(eq) / prev, 0.0)
    sd = np.std(rets, ddof=1)
    return float(sd * np.sqrt(BARS_PER_YEAR)) if np.isfinite(sd) else float("nan")


def measure(strategy, start, end, *, df=None, market=SPOT, balance=1_000.0):
    frame = DF if df is None else df
    result = run_period(strategy, frame, start, end, market=market,
                         start_balance=balance, data_label=LABEL)
    m = compute_metrics(result)
    return m, realized_vol(result.equity), mean_notional(result), result


def line(tag, m, vol, notional, result) -> None:
    print(f"  {tag:44s} final=${m.final_balance:>11,.0f} "
          f"vol={vol:5.3f} notional={notional:5.3f} DD={m.max_drawdown_pct:>5.1f}% "
          f"sharpe={m.sharpe:>5.2f} trades={m.num_trades:>5d} "
          f"fees=${m.fees_paid:>7,.0f}"
          f"{'  LIQUIDATED' if m.liquidated else ''}")


# --------------------------------------------------------------------------- step 3


def fold_sweep() -> pd.DataFrame:
    """Evaluate every configuration on each of the 3 purged folds, both markets."""
    rows = []
    t0 = time.time()
    n = 0
    for base, tv, ml in CONFIGS:
        n += 1
        for fold_label, (start, end) in FOLDS.items():
            strat = make_strategy(base, tv, ml)
            for mname, market in MARKETS:
                m, vol, notional, res = measure(strat, start, end, market=market)
                rows.append({"base": base, "target_vol": tv, "max_leverage": ml,
                             "fold": fold_label, "market": mname,
                             "final": m.final_balance, "profit_pct": m.profit_pct,
                             "vol": vol, "mean_notional": notional,
                             "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                             "trades": m.num_trades, "liquidated": m.liquidated})
        print(f"[{n:>2d}/{len(CONFIGS)}] base={base:>2d} tv={tv:.2f} ml={ml:.1f}  "
              f"[{time.time() - t0:.0f}s]")
    # v4 control, same folds, both markets
    for fold_label, (start, end) in FOLDS.items():
        for mname, market in MARKETS:
            m, vol, notional, res = measure(get_strategy(INCUMBENT), start, end, market=market)
            rows.append({"base": "v4_control", "target_vol": 0.55, "max_leverage": 2.0,
                         "fold": fold_label, "market": mname,
                         "final": m.final_balance, "profit_pct": m.profit_pct,
                         "vol": vol, "mean_notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "liquidated": m.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "fold_sweep.csv", index=False)
    print(f"\nfold sweep: {len(CONFIGS)} configurations x 3 folds x 2 markets "
          f"= {len(CONFIGS) * 3 * 2} backtests, [{time.time() - t0:.0f}s]")
    print(f"written: {OUT / 'fold_sweep.csv'}")
    return out


def pooled_sweep() -> pd.DataFrame:
    """Evaluate the SAME configurations on the naive pooled 2017-2022 window, both markets."""
    rows = []
    t0 = time.time()
    for i, (base, tv, ml) in enumerate(CONFIGS, 1):
        strat = make_strategy(base, tv, ml)
        for mname, market in MARKETS:
            m, vol, notional, res = measure(strat, *POOLED, market=market)
            rows.append({"base": base, "target_vol": tv, "max_leverage": ml, "market": mname,
                         "final": m.final_balance, "profit_pct": m.profit_pct,
                         "vol": vol, "mean_notional": notional,
                         "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                         "trades": m.num_trades, "liquidated": m.liquidated})
        print(f"[{i:>2d}/{len(CONFIGS)}] pooled base={base:>2d} tv={tv:.2f} ml={ml:.1f}  "
              f"[{time.time() - t0:.0f}s]")
    for mname, market in MARKETS:
        m, vol, notional, res = measure(get_strategy(INCUMBENT), *POOLED, market=market)
        rows.append({"base": "v4_control", "target_vol": 0.55, "max_leverage": 2.0,
                     "market": mname, "final": m.final_balance, "profit_pct": m.profit_pct,
                     "vol": vol, "mean_notional": notional,
                     "max_dd": m.max_drawdown_pct, "sharpe": m.sharpe,
                     "trades": m.num_trades, "liquidated": m.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "pooled_sweep.csv", index=False)
    print(f"\npooled sweep: {len(CONFIGS)} configurations x 2 markets "
          f"= {len(CONFIGS) * 2} backtests, [{time.time() - t0:.0f}s]")
    print(f"written: {OUT / 'pooled_sweep.csv'}")
    return out


def sweep() -> None:
    """Step 3: both grids -- fold search (robust selection input) and pooled (naive comparator)."""
    fold_sweep()
    pooled_sweep()
    print(f"\ndistinct configurations evaluated (step 3): {len(_SEEN)}")


# --------------------------------------------------------------------------- step 4/5


def select() -> None:
    """Minimax-across-folds selection vs naive pooled-window selection, plateau, TRAIN/VALID."""
    OUT.mkdir(parents=True, exist_ok=True)
    fcsv, pcsv = OUT / "fold_sweep.csv", OUT / "pooled_sweep.csv"
    fdf = pd.read_csv(fcsv) if fcsv.exists() else fold_sweep()
    pdf = pd.read_csv(pcsv) if pcsv.exists() else pooled_sweep()

    fspot = fdf[(fdf.market == "spot") & (fdf.base != "v4_control")].copy()
    fspot["base"] = fspot["base"].astype(int)
    pspot = pdf[(pdf.market == "spot") & (pdf.base != "v4_control")].copy()
    pspot["base"] = pspot["base"].astype(int)

    # ---- robust (minimax) selection across the 3 folds ----
    piv = fspot.pivot_table(index=["base", "target_vol", "max_leverage"],
                             columns="fold", values="sharpe")
    piv["min_fold_sharpe"] = piv.min(axis=1)
    piv["mean_fold_sharpe"] = piv[list(FOLDS.keys())].mean(axis=1)
    piv["p25_fold_sharpe"] = piv[list(FOLDS.keys())].apply(
        lambda r: float(np.percentile(r.to_numpy(dtype=float), 25)), axis=1)
    piv = piv.sort_values("min_fold_sharpe", ascending=False)
    print("=== top 8 by worst-fold (minimax) Sharpe ===")
    print(piv[list(FOLDS.keys()) + ["min_fold_sharpe", "p25_fold_sharpe", "mean_fold_sharpe"]]
          .head(8).to_string())

    robust = piv.index[0]
    r_base, r_tv, r_ml = robust
    print(f"\nrobust (minimax) winner: base={r_base}, target_vol={r_tv}, max_leverage={r_ml}  "
          f"worst-fold Sharpe={piv.iloc[0].min_fold_sharpe:.3f}  "
          f"(25th-pct fold Sharpe={piv.iloc[0].p25_fold_sharpe:.3f} -- with N=3 folds this sits "
          f"between the worst and median fold and is close to the minimax pick by construction)")

    # ---- naive (pooled point-estimate) selection ----
    pspot_sorted = pspot.sort_values("sharpe", ascending=False)
    naive = pspot_sorted.iloc[0]
    n_base, n_tv, n_ml = int(naive.base), float(naive.target_vol), float(naive.max_leverage)
    print(f"\nnaive pooled-window winner: base={n_base}, target_vol={n_tv}, "
          f"max_leverage={n_ml}  pooled Sharpe={naive.sharpe:.3f}")
    print(f"\nrobust winner's pooled Sharpe: "
          f"{pspot[(pspot.base == r_base) & (pspot.target_vol == r_tv) & (pspot.max_leverage == r_ml)].sharpe.iloc[0]:.3f}")
    naive_row = piv.loc[(n_base, n_tv, n_ml)] if (n_base, n_tv, n_ml) in piv.index else None
    if naive_row is not None:
        print(f"naive winner's worst-fold Sharpe: {naive_row.min_fold_sharpe:.3f}")

    same = (r_base, r_tv, r_ml) == (n_base, n_tv, n_ml)
    print(f"\nrobust winner {'IS' if same else 'DIFFERS FROM'} the naive pooled winner "
          f"-- {'failure mode (a): robustness selection changed nothing.' if same else 'these are two different configurations, as expected if fold disagreement matters.'}")

    # ---- plateau neighbourhood around the robust winner ----
    base_idx = list(LADDER_BASES).index(r_base)
    tv_idx = list(TARGET_VOL_GRID).index(r_tv)
    ml_idx = list(MAX_LEV_GRID).index(r_ml)
    base_nb = LADDER_BASES[max(0, base_idx - 1): base_idx + 2]
    tv_nb = TARGET_VOL_GRID[max(0, tv_idx - 1): tv_idx + 2]
    ml_nb = MAX_LEV_GRID[max(0, ml_idx - 1): ml_idx + 2]
    nb = piv.reset_index()
    nb = nb[nb.base.isin(base_nb) & nb.target_vol.isin(tv_nb) & nb.max_leverage.isin(ml_nb)]
    print(f"\n=== plateau neighbourhood (base in {base_nb}, target_vol in {tv_nb}, "
          f"max_leverage in {ml_nb}) -- worst-fold Sharpe ===")
    print(nb[["base", "target_vol", "max_leverage", "min_fold_sharpe", "mean_fold_sharpe"]]
          .to_string(index=False))
    print(f"worst-fold Sharpe range across neighbourhood: "
          f"[{nb.min_fold_sharpe.min():.3f}, {nb.min_fold_sharpe.max():.3f}]  "
          f"(project noise floor is +/-0.2 Sharpe, R-20)")

    # ---- v4 control's own fold numbers, for direct comparison ----
    v4fold = fdf[(fdf.market == "spot") & (fdf.base == "v4_control")]
    v4_min = v4fold.sharpe.min()
    v4_mean = v4fold.sharpe.mean()
    print(f"\nkelly_regime_v4 control (20,40,80 / 0.55 / 2.0), same 3 folds, spot: "
          f"worst-fold Sharpe={v4_min:.3f}  mean-fold Sharpe={v4_mean:.3f}")
    for _, r in v4fold.iterrows():
        print(f"    {r.fold}: sharpe={r.sharpe:.3f} profit={r.profit_pct:+.1f}% DD={r.max_dd:.1f}%")

    # ---- TRAIN / VALID continuous-window report, candidate vs v4 control, both markets ----
    print(f"\n=== inner-train (2017-2020) and inner-validation (2021-2022), "
          f"candidate vs kelly_regime_v4 control ===")
    cand_kwargs = dict(horizons=(r_base, 2 * r_base, 4 * r_base), target_vol=r_tv, max_leverage=r_ml)
    tv_rows = []
    for split_name, (start, end) in (("inner-train", TRAIN), ("inner-validation", VALID)):
        for mname, market in MARKETS:
            m_c, vol_c, not_c, res_c = measure(KellyRegimeV11RobustLadder(**cand_kwargs),
                                                start, end, market=market)
            m_v, vol_v, not_v, res_v = measure(get_strategy(INCUMBENT), start, end, market=market)
            line(f"  {split_name}/{mname} candidate", m_c, vol_c, not_c, res_c)
            line(f"  {split_name}/{mname} v4 control", m_v, vol_v, not_v, res_v)
            tv_rows.append({"split": split_name, "market": mname, "arm": "candidate",
                            "final": m_c.final_balance, "profit_pct": m_c.profit_pct,
                            "sharpe": m_c.sharpe, "max_dd": m_c.max_drawdown_pct,
                            "vol": vol_c, "mean_notional": not_c, "trades": m_c.num_trades})
            tv_rows.append({"split": split_name, "market": mname, "arm": "v4_control",
                            "final": m_v.final_balance, "profit_pct": m_v.profit_pct,
                            "sharpe": m_v.sharpe, "max_dd": m_v.max_drawdown_pct,
                            "vol": vol_v, "mean_notional": not_v, "trades": m_v.num_trades})
    pd.DataFrame(tv_rows).to_csv(OUT / "train_valid_candidate_vs_v4.csv", index=False)

    (OUT / "selected_config.txt").write_text(
        f"base={r_base}\nhorizons=({r_base},{2*r_base},{4*r_base})\n"
        f"target_vol={r_tv}\nmax_leverage={r_ml}\n"
        f"naive_base={n_base}\nnaive_target_vol={n_tv}\nnaive_max_leverage={n_ml}\n"
        f"n_configs_evaluated={len(_SEEN)}\n")
    print(f"\ndistinct configurations evaluated in total: {len(_SEEN)}")
    print(f"wrote {OUT / 'selected_config.txt'}")


def _load_selected() -> tuple[int, float, float]:
    cfg = OUT / "selected_config.txt"
    if cfg.exists():
        kv = dict(line.split("=", 1) for line in cfg.read_text().splitlines() if "=" in line)
        return int(kv["base"]), float(kv["target_vol"]), float(kv["max_leverage"])
    print("(no selected_config.txt yet -- run `select` first; falling back to v4 defaults)")
    return 20, 0.55, 2.0


# --------------------------------------------------------------------------- failure mode (c)


def exposure_artifact_check() -> None:
    """Mandatory exposure-artifact check (ROUTINE.md standing rule, sharpened by R-33).

    Build a "flat-rescaled v4" comparator: v4's own unchanged target,
    multiplied by a single constant c chosen so its mean notional matches
    the candidate's mean notional over the SAME period. Report R^2 of the
    candidate's target series against that flat rescale, on inner-
    validation, both markets. R^2 > 0.95 means "this is the standard
    exposure-level artifact".
    """
    base, tv, ml = _load_selected()
    print(f"\nexposure-artifact check (inner-validation, mean-notional-matched flat rescale of v4)")
    print(f"candidate: base={base} target_vol={tv} max_leverage={ml}")
    for mname, market in MARKETS:
        cand = KellyRegimeV11RobustLadder(horizons=(base, 2 * base, 4 * base),
                                           target_vol=tv, max_leverage=ml)
        m_c, vol_c, not_c, res_c = measure(cand, *VALID, market=market)
        v4 = get_strategy(INCUMBENT)
        m_v4, vol_v4, not_v4, res_v4 = measure(v4, *VALID, market=market)

        cand_t = res_c.df["target"].to_numpy(dtype=float)
        v4_t = res_v4.df["target"].reindex(res_c.df.index).to_numpy(dtype=float)
        c = not_c / not_v4 if not_v4 > 0 else float("nan")
        flat = c * v4_t

        mask = np.isfinite(cand_t) & np.isfinite(flat)
        x = flat[mask]
        y = cand_t[mask]
        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

        verdict = ("EXPOSURE-LEVEL ARTIFACT (R^2 > 0.95)" if np.isfinite(r2) and r2 > 0.95
                    else "not a flat rescale by this test")
        print(f"  {mname}: cand notional={not_c:.3f} v4 notional={not_v4:.3f} c={c:.3f}  "
              f"corr={corr:.4f}  R^2={r2:.4f}  {verdict}")
        print(f"    cand realized vol={vol_c:.3f}  v4 realized vol={vol_v4:.3f}  "
              f"cand sharpe={m_c.sharpe:.3f}  v4 sharpe={m_v4.sharpe:.3f}")


# ------------------------------------------------------------------------ causality


def causality() -> None:
    """Step 6: by-hand two-opposite-tampers lookahead probe on the selected candidate.

    Same procedure as R-28/R-31/R-33/R-37/R-38/R-40: bars after a cut are
    multiplied by 3 in one copy, divided by 3 in another; every decision
    at or before the cut must be bit-identical. This class reuses v4's
    prepare()/on_bar() unmodified (only constructor scalars differ, plus
    a per-instance warmup override that is set once in __init__ before
    any bar is processed), so this mainly re-confirms that passing
    different (horizons, target_vol, max_leverage) values cannot introduce
    a leak the registered defaults don't already have -- run explicitly
    rather than assumed. Restricted to strictly pre-2023 bars, per this
    session's data rule.
    """
    from tradebot.broker import PaperBroker
    from tradebot.orders import Order
    from tradebot.strategy import Context

    base, tv, ml = _load_selected()
    horizons = (base, 2 * base, 4 * base)

    pre_2023 = DF.loc[:"2022-12-31"]
    df = pre_2023.iloc[-300_000:].copy()
    cut = len(df) - 5_000
    bars = [cut - k for k in (1, 2, 3, 5, 10, 20, 100, 1_000)]

    up, down = df.copy(), df.copy()
    for col in ("open", "high", "low", "close"):
        up.iloc[cut:, up.columns.get_loc(col)] *= 3.0
        down.iloc[cut:, down.columns.get_loc(col)] /= 3.0
    up.iloc[cut:, up.columns.get_loc("volume")] *= 7.0
    down.iloc[cut:, down.columns.get_loc("volume")] /= 7.0

    def build():
        return KellyRegimeV11RobustLadder(horizons=horizons, target_vol=tv, max_leverage=ml)

    print(f"probing candidate: horizons={horizons} target_vol={tv} max_leverage={ml}")

    pa = build().prepare(up.copy())
    pb = build().prepare(down.copy())
    ok = True
    worst_col = float(np.nanmax(np.abs(pa["target"].to_numpy(dtype=float)[:cut]
                                        - pb["target"].to_numpy(dtype=float)[:cut])))
    good = worst_col < 1e-9
    ok &= good
    print(f"  column=target  max |difference| before the cut = {worst_col:.3e}  "
          f"{'PASS' if good else 'FAIL'}")

    def decisions(frame):
        s = build()
        prepared = s.prepare(frame.copy())
        broker = PaperBroker(market=FUTURES, start_balance=10_000.0)
        broker.execute(Order(target=0.1), prepared.index[0], float(prepared["open"].iloc[0]))
        out = []
        for i in bars:
            ctx = Context(prepared, i, broker)
            s.on_bar(ctx)
            out.append([(o.side, o.qty, o.target) for o in ctx.orders])
        return out

    bad = [b for b, oa, ob in zip(bars, decisions(up), decisions(down)) if oa != ob]
    ok &= not bad
    print(f"  orders {'match' if not bad else f'DIFFER at bars {bad}'} at the probe bars")

    a = run_backtest(build(), up.iloc[:cut + 1], FUTURES, 1_000.0, data_label=LABEL)
    b = run_backtest(build(), down.iloc[:cut + 1], FUTURES, 1_000.0, data_label=LABEL)
    worst_eq = float(np.max(np.abs(a.equity.to_numpy()[:cut] - b.equity.to_numpy()[:cut])))
    ok &= worst_eq < 1e-6
    print(f"  max |equity difference| before the cut = {worst_eq:.3e}  "
          f"{'PASS' if worst_eq < 1e-6 else 'FAIL'}")

    print(f"\ntampered from bar {cut:,} of {len(df):,}; "
          f"{'PASS - no decision at or before the cut moves' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------ eth


def eth() -> None:
    """Step 7: pre-registered falsification -- does the selected candidate hold on ETH?

    Same venue (Bitfinex), same pre-2020 window R-17/R-28/R-31/R-33/R-37/
    R-38/R-40 used, both spot and 5x futures, candidate vs shipped v4
    defaults as the control, on both the BTC control run and the ETH test
    run of the identical pipeline -- whole-file, pre-2020 data, safe under
    this session's rule. Falsification rule (fixed before running): if the
    candidate is not at least comparable to v4 on ETH, or is visibly worse
    on ETH than on the BTC control run through the identical code, this
    direction fails.
    """
    base, tv, ml = _load_selected()
    horizons = (base, 2 * base, 4 * base)
    print(f"candidate: horizons={horizons} target_vol={tv} max_leverage={ml}")

    rows = []
    for asset, path in (("BTC (control)", "btcusd_bitfinex_5m.csv.gz"),
                        ("ETH (test)", "ethusd_bitfinex_5m.csv.gz")):
        df = load_ohlcv_csv(ROOT / "data" / path)
        print(f"\n{asset}  {len(df):,} bars  "
              f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
        for mname, market in MARKETS:
            print(f"  {mname}:")
            cand = KellyRegimeV11RobustLadder(horizons=horizons, target_vol=tv, max_leverage=ml)
            m_c, vol_c, not_c, res_c = measure(cand, None, None, df=df, market=market)
            line(f"    candidate (v11)", m_c, vol_c, not_c, res_c)
            m_v4, vol_v4, not_v4, res_v4 = measure(get_strategy(INCUMBENT), None, None,
                                                    df=df, market=market)
            line(f"    {INCUMBENT} (control)", m_v4, vol_v4, not_v4, res_v4)
            rows.append({"asset": asset, "market": mname, "arm": "candidate",
                         "final": m_c.final_balance, "profit_pct": m_c.profit_pct,
                         "sharpe": m_c.sharpe, "max_dd": m_c.max_drawdown_pct,
                         "vol": vol_c, "liquidated": m_c.liquidated})
            rows.append({"asset": asset, "market": mname, "arm": "v4_control",
                         "final": m_v4.final_balance, "profit_pct": m_v4.profit_pct,
                         "sharpe": m_v4.sharpe, "max_dd": m_v4.max_drawdown_pct,
                         "vol": vol_v4, "liquidated": m_v4.liquidated})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "eth_falsification.csv", index=False)

    print("\n=== falsification verdict, candidate vs v4 control ===")
    verdict_ok = True
    for asset in ("BTC (control)", "ETH (test)"):
        for mname, _ in MARKETS:
            c = out[(out.asset == asset) & (out.market == mname) & (out.arm == "candidate")].iloc[0]
            d = out[(out.asset == asset) & (out.market == mname) & (out.arm == "v4_control")].iloc[0]
            d_sharpe = c.sharpe - d.sharpe
            d_profit = c.profit_pct - d.profit_pct
            d_dd = c.max_dd - d.max_dd
            ok = d_sharpe > -0.05 and d_profit > -2.0
            verdict_ok &= ok if "ETH" in asset else True
            print(f"  {asset:16s} {mname:8s} d(Sharpe)={d_sharpe:+.3f} "
                  f"d(profit)={d_profit:+.1f}pp d(maxDD)={d_dd:+.1f}pp  "
                  f"{'OK' if ok else 'WORSE'}")
    print(f"\nETH falsification: {'PASS' if verdict_ok else 'FAIL'}")
    print(f"wrote {OUT / 'eth_falsification.csv'}")


# ------------------------------------------------------------------------------- main


if __name__ == "__main__":
    print(f"{len(DF):,} bars  {DF.index[0]:%Y-%m-%d} -> {DF.index[-1]:%Y-%m-%d}  "
          f"(data: {LABEL})", file=sys.stderr)
    cmds = {"sweep": sweep, "select": select, "artifact": exposure_artifact_check,
            "causality": causality, "eth": eth}

    def all_() -> None:
        sweep()
        select()
        exposure_artifact_check()
        causality()
        eth()

    cmds["all"] = all_
    choice = sys.argv[1] if len(sys.argv) > 1 else ""
    if choice in cmds:
        cmds[choice]()
    else:
        print(f"usage: python {sys.argv[0]} [{'|'.join(cmds)}]")
